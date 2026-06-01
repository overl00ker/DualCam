#ifndef VIEWER_DIALOGS_H
#define VIEWER_DIALOGS_H

#include <QDialog>
#include <QJsonObject>

QDialog* makeProfileDialog(const QJsonObject& obj, QWidget* parent = nullptr);
QDialog* makeSurfaceDialog(const QJsonObject& obj, QWidget* parent = nullptr);

#endif // VIEWER_DIALOGS_H
